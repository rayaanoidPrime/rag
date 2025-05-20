from uuid import UUID

class AutoId:
    """
    Helper methods to detect txtai auto ids
    """

    @staticmethod
    def valid(uid):
        """
        Checks if uid is a valid auto id (UUID or numeric id).

        Args:
            uid: input id

        Returns:
            True if this is an autoid, False otherwise
        """

        # Check if this is a UUID
        try:
            # Ensure uid is string for UUID validation
            return isinstance(UUID(str(uid)), UUID)
        except ValueError:
            pass

        # Return True if this is numeric, False otherwise
        return isinstance(uid, int) or (isinstance(uid, str) and uid.isdigit())